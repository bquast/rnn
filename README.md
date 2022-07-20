rnn
===

[![License](https://img.shields.io/badge/license-GPLv3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![CRAN Version](https://www.r-pkg.org/badges/version/rnn)](https://cran.r-project.org/package=rnn)
[![R build status](https://github.com/bquast/rnn/workflows/R-CMD-check/badge.svg)](https://github.com/bquast/rnn/actions?workflow=R-CMD-check)
[![Coverage status](https://codecov.io/gh/bquast/rnn/branch/master/graph/badge.svg?token=bsdqgGdJNu)](https://app.codecov.io/gh/bquast/rnn)
[![Total RStudio Cloud Downloads](https://cranlogs.r-pkg.org/badges/grand-total/rnn?color=brightgreen)](https://cran.r-project.org/package=rnn)
[![RStudio Cloud Downloads](https://cranlogs.r-pkg.org/badges/rnn?color=brightgreen)](https://cran.r-project.org/package=rnn)

Implementation of a Recurrent Neural Network in R.


Demonstration
------------
![rnn demonstration](https://github.com/bquast/R-demo-GIFs/blob/master/rnn.gif)



Installation
------------
The **stable** version can be installed from [CRAN](https://cran.r-project.org/package=rnn) using:

```r
install.packages('rnn')
```

The **development** version, to be used **at your peril**, can be installed from [GitHub](https://github.com/bquast/rnn) using the `remotes` package.

```r
if (!require('remotes')) install.packages('remotes')
remotes::install_github('bquast/rnn')
```

Usage
-------------

Following installation, the package can be loaded using:

```r
library(rnn)
```

For general information on using the package, please refer to the help files.

```r
help('trainr')
help('predictr')
help(package='rnn')
```

There is also a long form vignette available using:

```r
vignette('rnn')
```


Additional Information
-----------------------

An overview of the changes is available in the NEWS file.

```r
news(package='rnn')
```

There is a dedicated website with information hosted on my [personal website](https://qua.st/).

https://qua.st/rnn/


Development
-------------
Development takes place on the GitHub page.

https://github.com/bquast/rnn/

Bugs can be filed on the issues page on GitHub.

https://github.com/bquast/rnn/issues
