AWS Infrastructure Update — EC2 Removal, S3 Retained

Summary:
We have decommissioned the m5.large EC2 instance (Ubuntu runtime) that previously hosted our trading system. All compute/network resources tied to EC2 have been removed, leaving S3 as the sole active AWS service for data persistence.

Actions Performed

    EC2 Instance Terminated

        The m5.large instance was terminated (not just stopped).

        This ensures no further hourly compute charges.

    EBS Volumes Deleted

        Root and attached Elastic Block Store volumes were deleted.

        Prevents continued storage billing.

    Snapshots Checked & Removed

        Old AMI/snapshots were deleted (unless intentionally archived).

        Cleared unused snapshot storage charges.

    Elastic IPs Released

        Any allocated but unused EIPs were released to avoid hourly charges.

    NAT Gateway / Load Balancers Verified

        None remain active; prevents hidden hourly charges.

Current State

    AWS Account remains active.

    Only S3 is in use.

    System artifacts (historical market data, engineered features, models) are expected to be saved to S3 buckets only.

    No compute charges should accrue; billing now consists of:

        S3 storage per GB per month.

        S3 request charges (PUT/GET).

        Data transfer out (first 100 GB/month free, then billed).

System Effects

    Local PC is now the primary runtime.

        Training, feature engineering, and pipelines run locally.

        Remote storage (S3) provides durability and offloads disk usage.

    Codex CLI / agents must adapt:

        Do not attempt to connect to EC2 (m5.large) — that environment no longer exists.

        Use local PC runtime for live trading system execution.

        Continue to interact with S3 buckets for historical data and model artifacts.

        IAM access keys for S3 remain valid; CLI/SDK should use these for reads/writes.

    Cost profile updated:

        No hourly EC2 charges.

        S3 will scale with dataset size; lifecycle rules should be applied (e.g., move old bars → Glacier/IA to minimize costs).

