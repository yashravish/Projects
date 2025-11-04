Mini example repo
-----------------

Structure
---------
- src/index.ts — simple modules and calls
- .github/workflows/ci.yml — CI steps for flow diagram

Try
---
```bash
node ../../dist/cli.js ingest src .github
node ../../dist/cli.js diagram sequence --question "Show main flow" --k 12
node ../../dist/cli.js diagram class --topic "index"
node ../../dist/cli.js diagram flow --question "CI pipeline"
```


