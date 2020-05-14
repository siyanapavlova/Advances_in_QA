# To transfer files to the dfki servers

jump_host=login.coli.uni-saarland.de
host=jones-5

echo -n "Enter your user name:";
read;
user=${REPLY}

echo "###########"
echo -n "Enter the remote file location in $host:";
read;
loc=${REPLY}

echo "###########"
echo "You want to retrieve file \"$loc\" on host \"$host\" hopping over \"$jump_host\" with username \"$user\" "
echo " onto your local machine at ./" 
echo "If yes, press enter to proceed further else CTRL-C to exit"
echo "###########"
read;

sleep 1

# and use command:


scp -o ProxyCommand="ssh $user@$jump_host nc $host 22" $user@$host:$loc ./

