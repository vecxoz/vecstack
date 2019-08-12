### Python 3.x

Since v0.4.0 vecstack provides official support for Python 3.5 and higher only,  
but still there is unofficial support for Python 2.7 and Python 3.4. See details below.  

The reason for these changes is global movement in Python 3.x direction.  
Vecstack depends on scikit-learn which has already stopped support for Python < 3.5.  
Scikit-learn v0.20.x is the last version supporting Python 2.7 and Python 3.4.  
Vecstack follows this direction as well.  
Please see [python3statement.org](https://python3statement.org/) for more details.  

### Unofficial support for Python 2.7 and Python 3.4

You can still install and run latest vecstack on Python 2.7 and Python 3.4.  
NOTE. It will require legacy versions of the following packages:   
* numpy<1.17
* scipy<1.3
* scikit-learn>=0.18,<0.21

There is a dedicated branch on GitHub called `py2` with appropriate requirements in `setup.py`.  
Installation:  

`pip install https://github.com/vecxoz/vecstack/archive/py2.zip`
