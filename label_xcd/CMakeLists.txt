
HAWC_ADD_EXECUTABLE (label3Class
  NO_PREFIX
  SOURCES labelClass_3class.cc
  USE_PROJECTS ${PROJECT_NAMES} hawcnest data-structures rng-service sweets
  USE_PACKAGES XCDF ROOT Boost XercesC)

HAWC_ADD_EXECUTABLE (label4Class
  NO_PREFIX
  SOURCES labelClass_4class.cc
  USE_PROJECTS ${PROJECT_NAMES} hawcnest data-structures rng-service sweets
  USE_PACKAGES XCDF ROOT Boost XercesC)


