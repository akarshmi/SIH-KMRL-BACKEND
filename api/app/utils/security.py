# from passlib.context import CryptContext

# pwd_context = CryptContext(
#     schemes=["bcrypt"],
#     deprecated="auto",
#     bcrypt__rounds=12
# )

# def hash_password(password: str) -> str:
#     # Truncate to 72 characters (bcrypt limitation) but keep as string
#     truncated_password = password[:72]
#     return pwd_context.hash(truncated_password)

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     # Truncate to 72 characters and keep as string
#     truncated_password = plain_password[:72]
#     return pwd_context.verify(truncated_password, hashed_password)


from api.app.config import pwd_context

def hash_password(password: str) -> str:
    """Hash password for storage"""
    return pwd_context.hash(password, scheme="bcrypt")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify user password"""
    return pwd_context.verify(plain_password, hashed_password)