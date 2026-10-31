# Provenance and scope

Original study code and frozen inputs: https://github.com/samarthraina/silent-influence at f8010273445adcf276c094cf5b228fc7bd0acf2a. The minimal reference tree preserves original paths and bytes for the recorded split inputs and prompt/scoring helpers. See reference/THIRD_PARTY_NOTICES.md for benchmark and generated-record terms. No weights are included.

The run_targeted.py and interventions.py files are byte-identical to the executed discovery files; their hashes are checked against both run metadata files. An old executor docstring says GPU execution had not been exercised in the authoring environment: that describes its initial local authoring, not the completed A100 runs packaged here.

Only the two completed discovery runs and their smoke controls are included. Server credentials, administration logs, and old automatic stop helpers are excluded. No reserved validation inference has been performed in this experiment. The split records earlier known exposure; joint exposure across collaborators must be audited before calling any future confirmation set unseen.
