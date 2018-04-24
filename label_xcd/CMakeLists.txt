
HAWC_ADD_EXECUTABLE (labelClass
  NO_PREFIX
  SOURCES labelClass.cc
  USE_PROJECTS ${PROJECT_NAMES} hawcnest data-structures rng-service sweets
  USE_PACKAGES XCDF ROOT Boost XercesC)

