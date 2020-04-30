include(findpathirdparty)

set(eigen_VERSION 3.2.5-bid3 CACHE INTERNAL "eigen version")
findpathirdparty(eigen eigen ${eigen_VERSION} noarch)

