# backend/api/app/config.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import cloudinary
import cloudinary.uploader
from passlib.context import CryptContext
import cloudinary.api
import os

# Load environment variables
load_dotenv()

# ---- Supabase ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    raise RuntimeError("Missing Supabase credentials in environment")

# ---- Cloudinary ----
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

# ---- Password Hashing ----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def configure_cloudinary():
    """Configure Cloudinary with environment variables"""
    
    # Get configuration from environment variables
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    api_key = os.getenv('CLOUDINARY_API_KEY') 
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
    
    if not all([cloud_name, api_key, api_secret]):
        raise ValueError(
            "Missing Cloudinary configuration. Please set:\n"
            "CLOUDINARY_CLOUD_NAME\n"
            "CLOUDINARY_API_KEY\n"
            "CLOUDINARY_API_SECRET\n"
            "in your environment variables or .env file"
        )
    
    # Configure Cloudinary
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True  # Always use HTTPS URLs
    )
    
    print(f"✅ Cloudinary configured for cloud: {cloud_name}")
    return True

def test_cloudinary_connection():
    """Test if Cloudinary is properly configured and accessible"""
    try:
        # Test by uploading a small test file
        result = cloudinary.uploader.upload(
            "data:text/plain;base64,SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            resource_type="raw",
            public_id="connection_test",
            folder="test"
        )
        
        # Clean up test file
        cloudinary.uploader.destroy("test/connection_test", resource_type="raw")
        
        print("✅ Cloudinary connection test successful")
        print(f"Test URL: {result.get('secure_url', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Cloudinary connection test failed: {str(e)}")
        return False

def get_optimized_upload_params(file_type: str, file_size: int):
    """Get optimized upload parameters based on file type and size"""
    
    params = {
        'use_filename': True,
        'unique_filename': True,
        'folder': 'documents',
        'secure': True
    }
    
    # Determine resource type
    if file_type.startswith('image/'):
        params['resource_type'] = 'image'
        # Add image optimizations
        params['quality'] = 'auto'
        params['fetch_format'] = 'auto'
    elif file_type == 'application/pdf':
        params['resource_type'] = 'image'  # PDFs handled as images
        params['format'] = 'pdf'
    else:
        params['resource_type'] = 'raw'
    
    # Add file size optimizations
    if file_size > 5 * 1024 * 1024:  # 5MB
        params['chunk_size'] = 6000000  # Upload in 6MB chunks
    
    return params