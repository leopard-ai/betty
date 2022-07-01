# Contributing

We welcome and greatly appreciate contributions from the community! You can
help make Betty even better by either

1. [Creating a PR](#create-a-pr) to fix a bug or add functionality.
2. [Reporting a bug](#report-a-bug) that you found in Betty.
3. [Requesting a feature](#request-a-feature) that you would like to
   see in Betty.

# Create a PR

To create a PR:

1. If applicable, make sure to write unit tests that exhaustively test your
   added code. Run all unit tests and ensure that there are no failures.
2. Submit a PR to the [main branch](.).
   If your PR fixes a bug, detail what the problem was and how it was fixed.
   If your PR adds code, include justification for why this code should be added.
3. The maintainers will discuss your PR and merge it into the dev branch if
   accepted. The dev branch will be periodically merged into the master branch.

### What kind of additional features are we looking for?

If there is an important metric that you believe is missing from Betty, we would
be grateful if you submitted a PR adding this metric. For other types of features,
ask yourself:

* Does this change add a new feature that will have a positive effect on the
  majority of Betty's users?
* Does this change make the codebase more confusing or difficult to deal with?
* Does this change add heavy or niche dependencies?

Feel free to [submit an issue](#request-a-feature) before making the
feature to see if your feature is something that the maintainers or other users
would want.

### Code Format

We use [black](https://black.readthedocs.io/en/stable/getting_started.html) to
format our code. Please use the latest stable version, and run
```bash
black .
```
in the root directory before creating a PR.

# Report a Bug

If you find a bug, please make an issue and include [BUG] in the title. In 
your issue, please give a description of how to reproduce the bug.

# Request a Feature

For any feature request, submit an issue with [REQUEST] in the title.
As part of the issue, describe why this would be beneficial for Betty and give
an example use case of how the feature would be used.

# Maintainers

If you have any questions, feel free to reach out to the maintainers:

* [Sang Keun Choe](https://github.com/sangkeun00) (sangkeuc (at) cs.cmu.edu)
* [Willie Neiswanger](https://github.com/willieneis) (neiswanger (at) cs.stanford.edu)
