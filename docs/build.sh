rm -rf _build
make html
firefox _build/html/index.html
open -a Safari _build/html/index.html
