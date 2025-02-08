# patchmgmt/aws_client.py
async def deploy_patch(kb_number):
    ssm = boto3.client('ssm')
    response = ssm.send_command(
        Targets=[{'Key':'InstanceIds','Values': get_ec2_targets()}],
        DocumentName='AWS-RunPatchBaseline',
        Parameters={'Operation': ['Install'], 'KBId': [kb_number]}
    )
    return response['CommandId']