# pkfinish

Optional compact-down last-CTA finish. The last producer for a 64-column tile combines original per-slot FP32 partials and writes BF16 outputs. The prior compact producer clears per-invocation tile counters. Source includes a remote-only exactness, graph replay and full-chain timing admission; original paths remain for unsupported and rejected cases. Dispatch is scoped to a private full-step graph capture. Compilation, activation, numerical equivalence and complete-engine benefit are not established by the source.
