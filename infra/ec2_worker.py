"""Provision / control the EC2 Spot worker for the challenge (idempotent).

  python infra/ec2_worker.py up        # role + SG + persistent Spot r7i.2xlarge (stop-on-interrupt) + idle alarm
  python infra/ec2_worker.py status
  python infra/ec2_worker.py stop|start
  python infra/ec2_worker.py run "<shell cmd>"   # via SSM RunCommand (no SSH), prints output

Everything is tagged project=amlc2026. Access is via SSM only (no inbound ports, no key pair).
"""
import json
import sys
import time

import boto3

REGION, PROJECT = "ap-south-1", "amlc2026"
BUCKET = "amlc2026-856608368454"
ROLE = PROFILE = "amlc2026-ec2-worker"
SG_NAME = "amlc2026-worker-sg"
INSTANCE_TYPE = "r7i.2xlarge"
VOLUME_GB = 150
TAGS = [{"Key": "project", "Value": PROJECT}]

iam = boto3.client("iam")
ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)
cw = boto3.client("cloudwatch", region_name=REGION)

USER_DATA = f"""#!/bin/bash
set -eux
dnf install -y python3.11 python3.11-pip git htop tmux
mkdir -p /opt/amlc && cd /opt/amlc
python3.11 -m venv .venv
echo 'ready' > /opt/amlc/BOOTSTRAPPED
"""


def ensure_role():
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                                                      "Action": "sts:AssumeRole"}]}
    try:
        iam.get_role(RoleName=ROLE)
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Tags=TAGS,
                        Description="AMLC2026 worker: SSM + project bucket only")
    iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    s3_policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{BUCKET}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": f"arn:aws:s3:::{BUCKET}/*"}]}
    iam.put_role_policy(RoleName=ROLE, PolicyName="project-bucket", PolicyDocument=json.dumps(s3_policy))
    try:
        iam.get_instance_profile(InstanceProfileName=PROFILE)
    except iam.exceptions.NoSuchEntityException:
        iam.create_instance_profile(InstanceProfileName=PROFILE, Tags=TAGS)
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE, RoleName=ROLE)
        time.sleep(10)  # IAM propagation


def ensure_sg():
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    sgs = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if sgs:
        return sgs[0]["GroupId"]
    sg = ec2.create_security_group(GroupName=SG_NAME, Description="AMLC2026 worker - no inbound, SSM only", VpcId=vpc,
                                   TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS}])
    return sg["GroupId"]  # default rules: no inbound, all outbound


def find_instance(states=("pending", "running", "stopping", "stopped")):
    r = ec2.describe_instances(Filters=[{"Name": "tag:project", "Values": [PROJECT]},
                                        {"Name": "instance-state-name", "Values": list(states)}])
    inst = [i for res in r["Reservations"] for i in res["Instances"]]
    return inst[0] if inst else None


def ensure_idle_alarm(instance_id):
    cw.put_metric_alarm(
        AlarmName=f"{PROJECT}-idle-stop-{instance_id}", Namespace="AWS/EC2", MetricName="CPUUtilization",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}], Statistic="Average", Period=300,
        EvaluationPeriods=6, Threshold=3.0, ComparisonOperator="LessThanThreshold", TreatMissingData="notBreaching",
        AlarmActions=[f"arn:aws:automate:{REGION}:ec2:stop"], Tags=TAGS,
        AlarmDescription="Stop the worker after 30 min below 3% CPU")


def up():
    existing = find_instance()
    if existing:
        print("already exists:", existing["InstanceId"], existing["State"]["Name"])
        return existing["InstanceId"]
    ensure_role()
    sg = ensure_sg()
    ami = ssm.get_parameter(Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")["Parameter"]["Value"]
    for attempt in range(6):
        try:
            r = ec2.run_instances(
                ImageId=ami, InstanceType=INSTANCE_TYPE, MinCount=1, MaxCount=1,
                IamInstanceProfile={"Name": PROFILE}, SecurityGroupIds=[sg], UserData=USER_DATA,
                InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
                    "SpotInstanceType": "persistent", "InstanceInterruptionBehavior": "stop"}},
                BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": VOLUME_GB, "VolumeType": "gp3",
                                                                          "DeleteOnTermination": True}}],
                MetadataOptions={"HttpTokens": "required"},
                TagSpecifications=[{"ResourceType": t, "Tags": TAGS + [{"Key": "Name", "Value": f"{PROJECT}-worker"}]}
                                   for t in ("instance", "volume", "spot-instances-request")])
            break
        except ec2.exceptions.ClientError as e:
            if "Invalid IAM Instance Profile" in str(e) and attempt < 5:
                time.sleep(10)
                continue
            raise
    iid = r["Instances"][0]["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    ensure_idle_alarm(iid)
    print("launched", iid)
    return iid


def run(cmd, timeout=3600):
    iid = find_instance(("running",))["InstanceId"]
    c = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript", TimeoutSeconds=timeout,
                         Parameters={"commands": [cmd], "executionTimeout": [str(timeout)]},
                         OutputS3BucketName=BUCKET, OutputS3KeyPrefix="ssm-logs")
    cid = c["Command"]["CommandId"]
    while True:
        time.sleep(3)
        try:
            inv = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            print(inv["StandardOutputContent"][-20000:])
            if inv["StandardErrorContent"].strip():
                print("STDERR:", inv["StandardErrorContent"][-5000:])
            print("status:", inv["Status"])
            return inv["Status"]


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "up":
        up()
    elif action == "status":
        i = find_instance(("pending", "running", "stopping", "stopped", "shutting-down"))
        print(i and (i["InstanceId"], i["State"]["Name"], i["InstanceType"], i.get("InstanceLifecycle")))
    elif action in ("stop", "start"):
        iid = find_instance()["InstanceId"]
        (ec2.stop_instances if action == "stop" else ec2.start_instances)(InstanceIds=[iid])
        print(action, iid)
    elif action == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3600)
