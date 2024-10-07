## Tips and tricks

### 2.2 Connect to your cluster

To easily login to your cluster via [AWS Systems Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html) we've included a script `easy-ssh.sh` that you can run like so, assuming `ml-cluster` is the name of your cluster:

```bash
./easy-ssh.sh ml-cluster
```

You'll need a few pre-requisites for this script:
* JQ: `brew install jq`
* aws cli
* `pcluster` cli
* [Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)

Once you've run the script you'll see the following output:

```
Instance Id: i-0096542c11ccb02b5
Os: ubuntu2004
User: ubuntu
Add the following to your ~/.ssh/config to easily connect:

cat <<EOF >> ~/.ssh/config
Host ml-cluster
  User ubuntu
  ProxyCommand sh -c "aws ssm start-session --target i-0095542c11ccb02b5 --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
EOF

Add your ssh keypair and then you can do:

$ ssh ml-cluster

Connecting to ml-cluster...

Starting session with SessionId: ...
root@ip-10-0-24-126:~#
```

1. Add your public key to the file `~/.ssh/authorized_keys`

2. Now paste in the lines from the output of to your terminal, this will add them to your `~/.ssh/config`.

```
cat <<EOF >> ~/.ssh/config
Host ml-cluster
  User ubuntu
  ProxyCommand sh -c "aws ssm start-session --target i-0095542c11ccb02b5 --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
EOF
```
3. Now you ssh in, assuming `ml-cluster` is the name of your cluster with:

```
ssh ml-cluster
```

