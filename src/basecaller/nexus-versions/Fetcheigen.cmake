include(findpathirdparty)

set(eigen_VERSION 3.3.9-bid4 CACHE INTERNAL "eigen version")
findpathirdparty(eigen eigen ${eigen_VERSION} noarch)
