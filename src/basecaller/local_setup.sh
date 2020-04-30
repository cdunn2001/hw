# Caution:
# This is only intended as a convenience tool if for whatever reason a developer needs to make
# a change affecting both this repository and pa-third-party/common.  This should be an uncommon
# need, and if you find yourself needing it often, please evaluate your situation to make sure
# it's truly necessary and appropriate.
#
# Please do not change the names/locations of submodules.  This repository has a .gitignore set
# up to keep any submodule usage local to the develop and not part of the centralized repository.
# If any naming changes then any submodule usage may accidentally get pushed up to the remote.
#
# Beware: If a local version is needed, it is entirely the responsibility of the developer to
#         make sure they are building with consistent versions.  The develop branch is checked
#         out by default, but this may not be correct in all cases.


git submodule add -b develop ssh://git@bitbucket.nanofluidics.com:7999/test/pa-third-party.git

git submodule add -b develop ssh://git@bitbucket.nanofluidics.com:7999/test/pa-common.git

git submodule add -b develop ssh://git@bitbucket.nanofluidics.com:7999/test/hw-mongo.git
