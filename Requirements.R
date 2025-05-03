# Requirements.R: list of CRAN packages to install via pak
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

pak::pkg_install(packages)
