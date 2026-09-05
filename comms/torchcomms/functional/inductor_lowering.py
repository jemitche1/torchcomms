# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Inductor lowerings for torchcomms collective operations."""

import logging

import torch


logger = logging.getLogger(__name__)


def _unpack_process_kernel(result):
    """Normalize ``ExternKernel.process_kernel`` output to the legacy 5-tuple

    PyTorch (pytorch/pytorch#189258, ~2026-07) changed ``process_kernel`` to
    return a frozen ``ProcessKernelResult`` dataclass instead of the positional
    5-tuple ``(example_output, tensor_args, non_tensor_args, unflatten_args,
    unbacked_bindings)``. The dataclass exposes the same values under the same
    names and in the same order. Accept both so we work across PyTorch
    versions.
    """
    if isinstance(result, tuple):
        return result
    return (
        result.example_output,
        result.tensor_args,
        result.non_tensor_args,
        result.unflatten_args,
        result.unbacked_bindings,
    )


try:  # noqa: C901
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering

    def _register_sideeffect_lowering(base_op_name: str, schema) -> None:
        """Register a _CollectiveKernel-based lowering for no-mutable-arg ops.

        Ops like barrier() and send() have no mutable tensor arg, so
        register_torchcomms_lowerings previously skipped custom lowering
        for them entirely, leaving them to the generic torch._inductor
        with_effects fallback (a plain ir.FallbackKernel).

        That matters because torch._inductor.comms.decide_global_ordering_of_comms
        - the pass whose entire job is to "enforce the ordering that's in
        the input graph" for communication ops - only considers a node for
        ordering if torch._inductor.utils.is_collective() recognizes it,
        which requires isinstance(node, ir._CollectiveKernel). A plain
        ir.FallbackKernel is invisible to that pass. So a mutating op like
        recv() (which does get a _CollectiveKernel via
        _register_functional_lowering/_register_inplace_lowering) is
        correctly kept in program order, while barrier()/send() are not -
        letting the scheduler reorder recv() ahead of them even though the
        traced program order was barrier() -> send() -> recv(). Point-to-
        point patterns (e.g. an even/odd send-then-recv vs. recv-then-send
        ring used specifically to avoid deadlock) depend on that order
        being preserved exactly, so this reordering causes a real
        deadlock, not just a performance regression.

        These ops' schemas say ``Tensor?`` (Optional) because whether they
        return a real tensor depends on the *value* of async_op at trace
        time, not on which op it is: barrier(async_op=True)/
        send(async_op=True) return a real (small, often opaque "work
        handle") tensor that a later .wait() call needs a genuine
        reference to, while the async_op=False case returns nothing.
        So the actual fake-traced example_output is inspected and handled
        either way: a real tensor gets a FixedLayout _CollectiveKernel (so
        is_collective() still recognizes it and a real handle is available
        for .wait()), and a genuine None gets a NoneLayout _CollectiveKernel
        (the same pattern _register_inplace_lowering uses for mutating
        ops, minus the mutation_outputs since there is nothing to mutate).

        process_kernel (which fake-traces the underlying op to determine
        its output) is called exactly once here, not once per branch. An
        earlier version of this function called it a second time in the
        None branch after already using it to decide which branch to take.
        That is wrong for any op whose real kernel does request/sequence-
        number bookkeeping even under fake tracing (as collective ops
        often do to pair up requests across ranks) - the extra trace call
        looks like a second real invocation and desyncs that bookkeeping
        from what actually runs, which showed up as a torchcomm_barrier()
        call in the same compiled function as torchcomm_reduce_scatter_v()
        corrupting the latter's result (wrong values, and on one rank a
        segfault) even though reduce_scatter_v's own lowering was
        untouched by this change.

        A version of this function that routed the real-tensor
        (async_op=True) case through a plain ir.FallbackKernel instead of
        _CollectiveKernel (leaving is_collective() unable to recognize
        barrier(async_op=True)/send(async_op=True)) was tried as a more
        conservative fix, since a prior _CollectiveKernel-based version of
        this branch had caused a hang in a test combining
        barrier(async_op=True) with reduce_scatter_v. That conservative
        version passed test_fullgraph_compile_send_recv in isolation
        (every count/dtype/async_op combination) but still hung on its
        async_op=True case specifically inside the full ~200-compile
        FullgraphCompileTest discover suite - i.e. the same
        works-in-isolation-fails-in-a-larger-run non-determinism seen with
        reduce_scatter_v, just for a different pair of ops. Given the sync
        case's deadlock was fixed by making it a _CollectiveKernel, the
        async case likely needs the same treatment for the same reason
        (is_collective() recognition -> decide_global_ordering_of_comms
        participation), so both branches build a _CollectiveKernel again
        here.
        """
        from torch._inductor.virtualized import V

        functional_op = getattr(torch.ops.torchcomms, base_op_name, None)
        if functional_op is None:
            return

        def _sideeffect_lowering(*args):
            logger.debug(f"Lowering side-effect-capable {base_op_name}")

            with V.graph.fake_mode:
                (
                    example_output,
                    tensor_args,
                    non_tensor_args,
                    unflatten_args,
                    unbacked_bindings,
                ) = _unpack_process_kernel(
                    ir._CollectiveKernel.process_kernel(functional_op.default, *args)
                )
            assert not unbacked_bindings, f"{functional_op} {unbacked_bindings}"

            if isinstance(example_output, torch.Tensor):
                for tensor_arg in tensor_args:
                    if not isinstance(tensor_arg, ir.TorchBindObject):
                        tensor_arg.realize()
                packed = ir._CollectiveKernel(
                    ir._CollectiveKernel.tensor_to_layout(example_output),
                    functional_op.default,
                    tensor_args,
                    non_tensor_args,
                    unflatten_args,
                )
                packed.outputs = [packed]
                return ir.TensorBox.create(packed)

            device = None
            for a in args:
                if isinstance(a, ir.TensorBox):
                    a.realize()
                    if device is None:
                        device = a.get_device()
            if device is None:
                # Ops like barrier() take no tensor arg at all, so there is
                # nothing above to derive a device from. If this happens to
                # be the first op lowered in the graph, V.graph.current_device
                # may also still be unset (nothing has claimed a device
                # context yet), so get_current_device_or_throw() would raise.
                # Fall back to any real tensor among the compiled function's
                # own inputs - the collective still runs on that device.
                device = V.graph.current_device
            if device is None:
                for graph_input in V.graph.graph_inputs.values():
                    if isinstance(graph_input, ir.IRNode):
                        try:
                            device = graph_input.get_device()
                        except Exception:
                            device = None
                        if device is not None:
                            break
            if device is None:
                # Last resort: this process is already bound to a specific
                # accelerator device (e.g. torch.xpu.set_device() in the
                # wrapper code that runs before any compiled kernel), so
                # ask for that directly instead of trying to infer it from
                # graph state.
                try:
                    device = torch.device(
                        torch.accelerator.current_accelerator().type,
                        torch.accelerator.current_device_index(),
                    )
                except Exception:
                    device = None
            if device is None:
                device = V.graph.get_current_device_or_throw()

            ir._CollectiveKernel(
                ir.NoneLayout(device=device),
                functional_op.default,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )

            return None

        register_lowering(functional_op.default)(_sideeffect_lowering)
        logger.info(f"Registered side-effect-capable lowering: {base_op_name}")

    def register_torchcomms_lowerings():
        from torchcomms.functional import collectives

        """Register all torchcomms collective lowerings with inductor."""
        if collectives is None:
            logger.warning("torchcomms.functional.collectives not available")
            return

        try:
            from torchcomms.functional.registry import _REGISTERED_COLLECTIVES

            # Register ops with the reinplace pass and create lowerings
            for base_op_name, info in _REGISTERED_COLLECTIVES.items():
                schema = info["param_schema"]
                if len(schema.mutable_params) > 0:
                    _register_with_reinplace_pass(base_op_name, schema)
                    _register_inplace_lowering(base_op_name, schema)
                    _register_functional_lowering(base_op_name, schema)
                else:
                    # No mutable tensor arg (e.g. barrier(), send(), or
                    # torchcommwindow_map_remote_tensor): still needs a
                    # _CollectiveKernel-based lowering so Inductor's
                    # scheduler recognizes it as a collective, regardless
                    # of whether this particular call ends up producing a
                    # real tensor output. See _register_sideeffect_lowering.
                    _register_sideeffect_lowering(base_op_name, schema)

            # Register lowering for functional wait_tensors
            _register_wait_tensors_lowering()

            logger.info("Registered torchcomms lowerings for torch.compile")
        except AttributeError as e:
            logger.warning(f"Failed to register torchcomms lowerings: {e}")

    def _register_with_reinplace_pass(base_op_name: str, schema) -> None:
        """Register functional/inplace op pair with the reinplace pass.

        This allows the reinplace pass to convert functional ops to inplace
        when ALL input tensors have no other uses.
        """
        try:
            from torch._inductor.fx_passes.reinplace import (
                inplaceable_ops,
                InplaceableOp,
            )

            functional_op = getattr(torch.ops.torchcomms, base_op_name, None)
            inplace_op = getattr(torch.ops.torchcomms, f"{base_op_name}_", None)

            if functional_op is None or inplace_op is None:
                return

            # Find all mutable tensor arg indices
            mutable_tensor_indices = []
            for i, p in enumerate(schema.input_params):
                if p.mutable and (
                    p.torch_type == "Tensor" or p.torch_type == "Tensor[]"
                ):
                    mutable_tensor_indices.append(
                        i + 1
                    )  # +1 for comm object at index 0

            if not mutable_tensor_indices:
                return

            inplaceable_ops[functional_op.default] = InplaceableOp(
                inplace_op.default,
                (
                    tuple(mutable_tensor_indices)
                    if len(mutable_tensor_indices) > 1
                    else mutable_tensor_indices[0]
                ),
            )
            logger.info(
                f"Registered reinplace: {base_op_name} -> {base_op_name}_ "
                f"(mutated_args={mutable_tensor_indices[0]})"
            )
        except ImportError:
            logger.debug("reinplace pass not available")

    def _register_inplace_lowering(base_op_name: str, schema) -> None:
        """Register lowering for the inplace op.

        The inplace op mutates tensors directly - no cloning needed.
        (Reinplace only converts to inplace when all tensors can be inplaced.)
        """
        from torch._inductor.virtualized import V

        inplace_op = getattr(torch.ops.torchcomms, f"{base_op_name}_", None)
        if inplace_op is None:
            return

        # Get indices of mutable tensor args (offset by 1 for the comm object)
        mutable_indices = []
        for i, p in enumerate(schema.input_params):
            if p.mutable and (p.torch_type == "Tensor" or p.torch_type == "Tensor[]"):
                mutable_indices.append(i + 1)

        def _inplace_lowering(*args):
            logger.debug(f"Lowering inplace {base_op_name}_")

            # Get the mutable tensors from args
            mutable_tensors = []
            for idx in mutable_indices:
                if idx < len(args):
                    tensor_arg = args[idx]
                    if isinstance(tensor_arg, ir.TensorBox):
                        mutable_tensors.append(tensor_arg)
                    elif isinstance(tensor_arg, (list, tuple)):
                        mutable_tensors.extend(
                            t for t in tensor_arg if isinstance(t, ir.TensorBox)
                        )

            if not mutable_tensors:
                logger.warning(f"No mutable tensors for {base_op_name}_")
                return None

            # Realize and mark tensors as mutated
            device = None
            for tensor in mutable_tensors:
                tensor.realize()
                V.graph.mark_buffer_mutated(tensor.get_name())
                if device is None:
                    device = tensor.get_device()

            # Process kernel args
            with V.graph.fake_mode:
                (
                    _example_output,
                    tensor_args,
                    non_tensor_args,
                    unflatten_args,
                    unbacked_bindings,
                ) = _unpack_process_kernel(
                    ir._CollectiveKernel.process_kernel(inplace_op.default, *args)
                )
            assert not unbacked_bindings, f"{inplace_op} {unbacked_bindings}"

            # Create the collective kernel
            packed = ir._CollectiveKernel(
                ir.NoneLayout(device=device),
                inplace_op.default,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )

            # Set up mutation_outputs
            packed.mutation_outputs.extend(
                [
                    ir.MutationOutput(ir.NoneLayout(device=device), buf, packed)
                    for buf in mutable_tensors
                ]
            )
            packed.alias_names.extend([t.get_name() for t in mutable_tensors])

            # Return the mutable tensors
            if len(mutable_tensors) == 1:
                return mutable_tensors[0]
            return mutable_tensors

        register_lowering(inplace_op.default)(_inplace_lowering)
        logger.info(f"Registered inplace lowering: {base_op_name}_")

    def _register_functional_lowering(base_op_name: str, schema) -> None:
        """Register lowering for the functional op.

        Uses FallbackKernel with the functional op directly.
        The functional op is non-mutable (returns new tensors), so FallbackKernel can handle it.
        """
        import torch.utils._pytree as pytree

        functional_op = getattr(torch.ops.torchcomms, base_op_name, None)

        if functional_op is None:
            return

        def _functional_lowering(*args):
            logger.debug(f"Lowering functional {base_op_name} with {len(args)} args")

            # Use FallbackKernel with the functional op
            # The functional op returns new tensors, so it's not mutable
            def wrap_tensors(x):
                return ir.TensorBox.create(x) if isinstance(x, ir.IRNode) else x

            result = pytree.tree_map(
                wrap_tensors,
                ir._CollectiveKernel.create_out_of_place(functional_op.default, *args),
            )

            return result

        register_lowering(functional_op.default)(_functional_lowering)
        logger.info(f"Registered functional lowering: {base_op_name}")

    def _register_wait_tensors_lowering() -> None:
        """Register lowerings for both functional and inplace wait_tensors."""
        from torch._inductor.virtualized import V

        # === REINPLACE PASS REGISTRATION ===
        # Allow reinplace to convert functional -> inplace when tensors have no other uses.
        try:
            from torch._inductor.fx_passes.reinplace import (
                inplaceable_ops,
                InplaceableOp,
            )

            functional_op = torch.ops.torchcomms.torchcomm_wait_tensors
            inplace_op = torch.ops.torchcomms.torchcomm_wait_tensors_

            # wait_tensors takes a list of tensors at index 0
            # All tensors in the list are mutated
            inplaceable_ops[functional_op.default] = InplaceableOp(
                inplace_op.default,
                0,  # arg index 0 is the tensor list
            )
            logger.info(
                "Registered reinplace: torchcomm_wait_tensors -> torchcomm_wait_tensors_"
            )
        except ImportError:
            logger.debug("reinplace pass not available for wait_tensors")

        # === FUNCTIONAL LOWERING ===
        # Use FallbackKernel to create new output tensors that depend on the wait.
        # FallbackKernel properly handles the op call and creates output buffers.
        def _wait_tensors_functional_lowering(*args):
            import torch.utils._pytree as pytree

            logger.debug(
                f"Lowering functional torchcomms.torchcomm_wait_tensors with {len(args)} args"
            )

            # The op takes a list of tensors as the first argument
            # Inductor may pass this as a list or as individual tensors
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                inputs = list(args[0])
            else:
                inputs = list(args)

            if not inputs:
                return []

            # Flatten to get individual TensorBox objects
            flat_inputs = []
            for inp in inputs:
                if isinstance(inp, ir.TensorBox):
                    flat_inputs.append(inp)
                elif isinstance(inp, (list, tuple)):
                    flat_inputs.extend(t for t in inp if isinstance(t, ir.TensorBox))

            if not flat_inputs:
                return []

            logger.info(f"  - Processing {len(flat_inputs)} TensorBox inputs")

            # Use FallbackKernel to create new output tensors
            # Pass flat_inputs as a list since the op signature is Tensor[] -> Tensor[]
            def wrap_tensors(x):
                return ir.TensorBox.create(x) if isinstance(x, ir.IRNode) else x

            result = pytree.tree_map(
                wrap_tensors,
                ir.FallbackKernel.create(
                    torch.ops.torchcomms.torchcomm_wait_tensors.default,
                    flat_inputs,
                ),
            )

            logger.debug(f"  - Created FallbackKernel result: {type(result)}")
            return result

        register_lowering(torch.ops.torchcomms.torchcomm_wait_tensors.default)(
            _wait_tensors_functional_lowering
        )
        logger.info("Registered functional wait_tensors lowering")

        # === INPLACE LOWERING ===
        # Used when reinplace pass converts functional -> inplace.
        # Uses _WaitKernel for proper wait semantics.
        # Now returns the input tensors to match the updated op signature.
        def _wait_tensors_inplace_lowering(*args):
            # The op takes a list of tensors as the first argument
            # Inductor may pass this as a list or as individual tensors
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                inputs = list(args[0])
            else:
                inputs = list(args)

            logger.debug(
                f"Lowering inplace torchcomms.torchcomm_wait_tensors_ with {len(inputs)} tensors"
            )

            if not inputs:
                return []

            # Flatten to get individual TensorBox objects
            flat_inputs = []
            for inp in inputs:
                if isinstance(inp, ir.TensorBox):
                    flat_inputs.append(inp)
                elif isinstance(inp, (list, tuple)):
                    flat_inputs.extend(t for t in inp if isinstance(t, ir.TensorBox))

            if not flat_inputs:
                return []

            # Realize all inputs and mark as mutated
            device = None
            for inp in flat_inputs:
                inp.realize()
                V.graph.mark_buffer_mutated(inp.get_name())
                if device is None:
                    device = inp.get_device()

            # Create wait kernel using the inplace op
            with V.graph.fake_mode:
                (
                    _example_output,
                    tensor_args,
                    non_tensor_args,
                    unflatten_args,
                    unbacked_bindings,
                ) = _unpack_process_kernel(
                    ir._WaitKernel.process_kernel(
                        torch.ops.torchcomms.torchcomm_wait_tensors_.default,
                        flat_inputs,
                    )
                )
            assert not unbacked_bindings

            packed = ir._WaitKernel(
                ir.NoneLayout(device=device),
                torch.ops.torchcomms.torchcomm_wait_tensors_.default,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )

            # Add MutationOutput for each input tensor to register with graph
            for inp in flat_inputs:
                packed.mutation_outputs.append(
                    ir.MutationOutput(ir.NoneLayout(device=device), inp, packed)
                )
            packed.alias_names.extend([t.get_name() for t in flat_inputs])

            # Return the input tensors (they are mutated in-place)
            return flat_inputs

        register_lowering(torch.ops.torchcomms.torchcomm_wait_tensors_.default)(
            _wait_tensors_inplace_lowering
        )
        logger.info(
            "Registered inplace wait_tensors lowering for %s",
            torch.ops.torchcomms.torchcomm_wait_tensors_.default,
        )

except ImportError:
    logger.info("torch._inductor not available, skipping torchcomms lowerings")

    def register_torchcomms_lowerings():
        pass
