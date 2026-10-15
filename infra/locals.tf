data "aws_caller_identity" "current" {}

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  claim_bucket_name = join("-", [
    var.project_name,
    var.environment,
    lower(var.initials),
    data.aws_caller_identity.current.account_id
  ])
}