# Requirements.R
# List all CRAN packages to install
packages <- c(
  'readr',
  'readxl',
  'polycor',
  'psych',
  'lavaan',
  'simstudy',
  'mokken',
  'semTools',
  'lordif'
)

# Install via pak for parallel, binary installs
pak::pkg_install(packages)
