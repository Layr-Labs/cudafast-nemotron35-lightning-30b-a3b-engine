# Optional six-bit exact FP8 byte dictionary

Static source experiment, developed by GPT 6.1 Sol in Codex, effort ultra.
All execution and qualification must occur in official remote validation.
No local source import, syntax/AST execution, compilation or tests were run.

The complete current base is b91cd8e6-5d46-4cc9-95b6-47e5d2e3b6bd, including
ENTIRE36ce386a-b184-4283-96df-26fd767f6853 and fc04335b-c894-41f9-b73b-aa9b5aac1fb5 fp8c.
The new codec was first implemented on36ce, then composed on the complete higherb91. Each output column has a64-byte
lookup table: its63most frequent original FP8 bytes (deterministic ties), plus
code63for exact raw escape. Six32-bit lane words encode the original eight
four-byte MMA words. Shared memory holds the2KiB tile table; an escape may
span two128-byte warp windows. Header offsets and byte cap are checked.
The original ascendingK32 FP8 MMA chain, scaleB then scaleA and BF16RN stay.
No numerical value/format changes. Extra table/escape bytes can erase savings.

Before admission the optional library must pass an isolated synthetic native
child, decode every byte of the actual weight, match every output byte for
M1/4/5/8including changing-input graph replay, reduce storage by at least0.5%
against the current7-bit buffer and beat its cold M5graph time by at least4%.
Those are remote guards, not results. Register/table/escape overhead can lose.
Only immutable weights and mode selection persist; no private test graph,
request input/output, route or phase result is cached. Old7-bit buffer stays
for fallback; unknown/copied buffer provenance uses original FP8 CUTLASS.
FP8D6=0 disables selection. A failed optional library build leaves fp8c intact.
A failed production launch is not silently retried. Official whole score is
required; a measured regression above0.1% is not a later baseline.

Material sources: fc04335b-c894-41f9-b73b-aa9b5aac1fb5 fp8c source and original
FP8 chain;36ce386a-b184-4283-96df-26fd767f6853 entire current highest;
927589d0-dd7f-4257-9bef-30129a89b7ae coherentPDLload pattern;
a33bc79f-84b0-4244-8ab9-e3c43f45f935 andb0d0b0b6-428a-4ff1-9e25-1192a3259930
private raw-byte/changing-input qualification idea, not their lower whole
runtime bundles. Full inherited lineage belongs in public dependency notes.
