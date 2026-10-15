output "claim_bucket_name" {
  description = "S3 bucket used for claims, policies, and results."
  value       = aws_s3_bucket.claims.bucket
}

output "claim_bucket_arn" {
  description = "ARN of the claim-processing S3 bucket."
  value       = aws_s3_bucket.claims.arn
}

output "claims_prefix" {
  value = "claims/"
}

output "policies_prefix" {
  value = "policies/"
}

output "results_prefix" {
  value = "results/"
}