api directory
=============

This directory has two purposes

1. to define the API data structures that are used as JSON objects in the REST API
2. to render documentation on the API in html format and create a linkable target for the application.

Documentation rendering flow
----------------------------

The documentation is taken from three sources

/doc/kes-paws-rest-api.md   - Human written markdown describing the general pa-ws operation and examples of usage
/doc/kes-paws-rest-api.yaml - Human written "Swagger" formatted description of the endpoints
/src/pa-ws/api/*Object.h    - Human written C++ code for the data payloads

When `make` is run in this directory (TODO: convert to cmake workflow), the C++ *.h files are parsed using a script to
generate an additional Swagger formatted description of the endpoints. And then the master kes-paws-rest-api.yaml and
this kes-paws-rest-api-swagger.yaml files are contenated and converted to html (kes-paws-rest-api-swagger.html).

When cmake is run in this directory, the kes-paws-rest-api-swagger.html file is converted to a static string in
a c++ file and built into a linkable cmake `libPaWsApi` target.  When the application is linked to `libPaWsApi`,
there will be a `std::string GetApiString()` function available that returns the entire contents of the html.

The pa-ws application uses this to return the html from its /api endpoint.

TODO:
  This directory does an "in source" build of the html files, which is undesirable. This documentation build
  should be moved in to the actual cmake build directory. It's just a bit messy and it was easier to 
  mock up as a Makefile flow.  The task at hand is to port the Makefile contents into the CMakeLists.txt, 
  with `add_custom_command`s as needed.