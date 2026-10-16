# V2 initial full-run failure

Published training commit: `f58964c46ba22c5d97002de2fff06412f570b019`.
The formal Bridge run recorded step 1 with loss 1.0093326568603516, then
failed the synthesized RGB finite/range assertion before the next recorded step.
No validation metrics or completed experiment results exist. DACL never started.
The failed sequence has exited; no retry was launched.

The original assertion did not log extrema, so the magnitude and precise
source of the range violation cannot be recovered from this run. Floating
point jitter/rotation roundoff is a suspected cause, not a verified diagnosis.
The prepared fix permits only 1e-6 endpoint roundoff, clamps that roundoff
to [0,1], and reports extrema for larger violations while retaining finite
checks. Clean targets, synthesis parameters, architecture and loss are unchanged.
Only static syntax validation was performed; no extra training or forward test.
A new tracked run plan and published commit are required before any rerun.
