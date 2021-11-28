# https://ken2blog.tokyo/study/%E3%80%90windows%E5%90%91%E3%81%91%E3%80%91google-colaboratory-90%E5%88%86%E3%82%BB%E3%83%83%E3%82%B7%E3%83%A7%E3%83%B3%E5%88%87%E3%82%8C%E5%AF%BE%E7%AD%96/

set /P URL="input your URL: "
echo %DATE% %TIME% start connecting...
 
for /L %%i in (1, 1, 12) do (
  start %URL%
  echo %%i step connected
  timeout /nobreak 3600
)