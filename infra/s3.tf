resource "aws_s3_bucket" "claims" {
  bucket = local.claim_bucket_name
}

resource "aws_s3_bucket_ownership_controls" "claims" {
  bucket = aws_s3_bucket.claims.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "claims" {
  bucket = aws_s3_bucket.claims.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "claims" {
  bucket = aws_s3_bucket.claims.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "claims" {
  bucket = aws_s3_bucket.claims.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}