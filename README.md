rnn
=======
[![CRAN Version](http://www.r-pkg.org/badges/version/rnn)](https://cran.r-project.org/package=rnn)
[![Total RStudio Cloud Downloads](http://cranlogs.r-pkg.org/badges/grand-total/rnn?color=brightgreen)](https://cran.r-project.org/package=rnn)
[![RStudio Cloud Downloads](http://cranlogs.r-pkg.org/badges/rnn?color=brightgreen)](https://cran.r-project.org/package=rnn)
[![Travis-CI Build Status](https://travis-ci.org/bquast/rnn.png?branch=master)](https://travis-ci.org/bquast/rnn)
[![Coverage Status](https://coveralls.io/repos/bquast/rnn/badge.svg?branch=master)](https://coveralls.io/r/bquast/rnn?branch=master)
[![License](http://img.shields.io/badge/license-GPLv3-brightgreen.svg)](http://www.gnu.org/licenses/gpl-3.0.html)

Implementation of a Recurrent Neural Network in R.

Installation
------------
The **stable** version can be installed from [CRAN](https://cran.r-project.org/package=rnn) using:

```r
install.packages('rnn')
```

The **development** version, to be used **at your peril**, can be installed from [GitHub](http://github.com/bquast/rnn) using the `devtools` package.

```r
if (!require('devtools')) install.packages('devtools')
devtools::install_github('bquast/rnn')
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

There is also a long form [vignette](https://cran.r-project.org/package=rnn/vignettes/rnn.html) available using:

```r
vignette('rnn')
```


Additional Information
-----------------------

An overview of the changes is available in the NEWS file.

```r
news(package='rnn')
```

There is a dedicated website with information hosted on my [personal website](http://qua.st/).

http://qua.st/rnn


Development
-------------
Development takes place on the GitHub page.

http://github.com/rnn/

Bugs can be filed on the issues page on GitHub.

https://github.com/bquast/rnn/issues
