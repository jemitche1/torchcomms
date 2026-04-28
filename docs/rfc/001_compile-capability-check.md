# RFC: Enabling Compiler Feature Checks for Torchcomms

## Summary

Currently, `is_torch_compile_supported_and_enabled()` is driven by:
- `TORCHCOMMS_PATCH_FOR_COMPILE=1`
- a minimum PyTorch version of `2.12`
- or `TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT=1` as an override.

but it does not validate if the runtime provides the compiler with the neccessary features required by torchcomms. This RFC proposes to replace environment variables as the source of truth for compile enablement with a capability check, while retaining them as a user option.




## Problem

`torchcomms.functional.__init__.py` only checks environment variables and a version.

But the compile path depends on a number of compiler related features:

- collective registration with compile support in `registry.py`
- autograd registration for generated ops in `registry.py`

- dynamo integration by `register_with_dynamo()` in `dynamo.py`

- inductor integration by `register_torchcomms_lowerings()` in `inductor_lowering.py`

Because of this discrepancy, the current compile support could be “enabled” without checking if the runtime actually has the features the torchcomms compile path uses.


## Proposal
 An enhanced compile support API, which retains the boolean API for comptibility.

### Proposed API

```python

class CompileSupport:
    user_opt_in: bool
    version_ok: bool
    supports_fake: bool
    supports_autograd: bool
    supports_dynamo_integration: bool
    supports_inductor_integration: bool


def get_compile_support() -> CompileSupport:

def is_torch_compile_supported_and_enabled() -> bool:
    return get_compile_support().enabled
```
#### Description
* user_opt: bool \
Indicates whether compile support is enabled by the user for torchcomms

* version_ok: bool\
Indicates whether the runtime satisfies the configured version

* supports_fake: bool \
Indicates whether fake/meta support is available in the runtime

* supports_autograd: bool \
Indicates whether the runtime provides autograd registration support needed for torchcomms

* supports_dynamo_integration: bool \
Indicates whether the torchcomms dynamo integration is available

* supports_inductor_integration: bool \
Indicates whether the torchcomms inductor lowering is available
