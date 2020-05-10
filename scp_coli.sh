jump_host=login.coli.uni-saarland.de
host=jones-5

#filepath=/local/simonp/
# and use command:


if [ -f "$1" ]
 then
   scp -o ProxyCommand="ssh simonp@$jump_host nc $host 22" simonp@$host:$1 $2
fi
