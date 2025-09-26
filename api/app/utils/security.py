from passlib.context import CryptContext

# Initialize password context with bcrypt scheme
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)

def hash_password(password: str) -> str:
    """Hash the password and ensure it's truncated to 72 characters (bcrypt limitation)."""
    if not password:
        raise ValueError("Password cannot be empty")
    
    truncated_password = password[:72]  # Truncate to bcrypt's max length
    return pwd_context.hash(truncated_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify if the plain password matches the stored hash."""
    return pwd_context.verify(plain_password, hashed_password)
