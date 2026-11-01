from pathlib import Path
import json
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication
from rest_framework.authtoken.models import Token

from .models import UserProfile
from .rag import (
    ask_rag,
    load_backend_document,
    DOCUMENT_DIR
)
# GET USER ROLE
def get_user_role(user):

    # Superuser
    if user.is_superuser:
        return "superadmin"

    try:
        profile = user.profile
    except UserProfile.DoesNotExist:
        return None

    # Database role: support
    # Application role: support_admin
    if profile.role == "support":
        return "support_admin"
    # Database role: product
    # Application role: product_admin
    if profile.role == "product":
        return "product_admin"
    # Database role: general
    # Application role: general_user
    if profile.role == "general":
        return "general_user"
    return None
# LOGIN
@csrf_exempt
def login_view(request):
    if request.method != "POST":
        return JsonResponse(
            {
                "error": "Only POST method is allowed."
            },
            status=405
        )
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse(
            {
                "error": "Invalid JSON."
            },
            status=400
        )
    username = data.get("username")
    password = data.get("password")
    # VALIDATE LOGIN FIELDS
    if not username or not password:
        return JsonResponse(
            {
                "error": "Username and password are required."
            },
            status=400
        )
    # AUTHENTICATE DJANGO USER
    user = authenticate(
        username=username,
        password=password
    )
    if user is None:
        return JsonResponse(
            {
                "error": "Invalid credentials."
            },
            status=400
        )
    # GET APPLICATION ROLE
    role = get_user_role(user)
    if role is None:
        return JsonResponse(
            {
                "error": "User profile is not configured."
            },
            status=400
        )

    # SUPPORT LEVEL
    support_level = None
    if role == "support_admin":
        try:
            support_level = user.profile.support_level
        except UserProfile.DoesNotExist:
            return JsonResponse(
                {
                    "error": "Support user profile is not configured."
                },
                status=400
            )
        if not support_level:
            return JsonResponse(
                {
                    "error": (
                        "Support level is not configured "
                        "for this support admin."
                    )
                },
                status=400
            )
    # CREATE / GET TOKEN
    token, created = Token.objects.get_or_create(
        user=user
    )
    # LOGIN SUCCESS
    return JsonResponse(
        {
            "username": user.username,
            "role": role,
            "support_level": support_level,
            "is_superuser": user.is_superuser,
            "token": token.key
        },
        status=200
    )
# UPLOAD / INDEX DOCUMENT
@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def index_backend_document(request):

    user = request.user
    uploaded_file = request.FILES.get("file")
    raw_doctype = request.data.get("doctype")
    # 1. CHECK FILE
    if not uploaded_file:
        return Response(
            {
                "error": "PDF file is required."
            },
            status=400
        )
    # 2. CHECK & NORMALIZE DOCUMENT TYPE
    if not raw_doctype:
        return Response(
            {
                "error": "Document type is required."
            },
            status=400
        )
    normalized_type = str(raw_doctype).lower().strip()
    if normalized_type in ["general", "product", "product_knowledge"]:
        doctype = "general"
    elif normalized_type in ["technical", "technical_knowledge", "support"]:
        doctype = "technical"
    else:
        doctype = normalized_type

    if doctype not in ["general", "technical"]:
        return Response(
            {
                "error": (
                    "Invalid document type. "
                    "Use 'general' or 'technical'."
                )
            },
            status=400
        )
    # 3. GET USER ROLE
    user_role = get_user_role(user)
    if user_role is None:
        return Response(
            {
                "error": "User profile is not configured."
            },
            status=403
        )
    # 4. PERMISSION CHECK
    # SUPERADMIN
    if user_role == "superadmin":
        allowed_doctypes = [
            "general",
            "technical"
        ]

    # SUPPORT ADMIN
    elif user_role == "support_admin":
        allowed_doctypes = [
            "technical"
        ]
    # PRODUCT ADMIN
    elif user_role == "product_admin":
        allowed_doctypes = [
            "general"
        ]
    # GENERAL USER
    elif user_role == "general_user":
        return Response(
            {
                "error": (
                    "General user does not have "
                    "permission to upload documents."
                )
            },
            status=403
        )

    else:
        return Response(
            {
                "error": "Invalid user role."
            },
            status=403
        )
    # Check selected knowledge base
    if doctype not in allowed_doctypes:
        return Response(
            {
                "error": (
                    "You do not have permission to upload "
                    f"documents to the {doctype} knowledge base."
                )
            },
            status=403
        )
    # 5. VALIDATE PDF
    if not uploaded_file.name.lower().endswith(".pdf"):
        return Response(
            {
                "error": "Only PDF files are allowed."
            },
            status=400
        )
    # 6. SELECT DOCUMENT DIRECTORY (AUTOMATICALLY CREATED)
    document_dir = DOCUMENT_DIR / doctype
    document_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    # 7. SAFE FILE NAME
    safe_name = Path(
        uploaded_file.name
    ).name
    temp_path = document_dir / ".uploading.tmp"
    final_path = document_dir / safe_name
    try:

        # SAVE TEMPORARY FILE
        with open(
            temp_path,
            "wb"
        ) as destination:

            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        # FIND EXISTING PDFs
        existing_files = list(
            document_dir.glob("*.pdf")
        )
        # REMOVE OLD PDF
        for old_file in existing_files:
            # If the old file is the same path,
            # remove it before replacing it.
            try:
                old_file.unlink()
            except PermissionError:
                raise PermissionError(
                    f"The existing {doctype} PDF is locked "
                    "by Windows, OneDrive, or another application."
                )
        # STORE NEW PDF
        temp_path.rename(
            final_path
        )
        print(
            f"New {doctype} PDF stored:",
            final_path
        )
        # CREATE / REFRESH RAG INDEX
        chunk_count = load_backend_document(
            doctype
        )
        print(
            f"{doctype} RAG indexing completed."
        )
        # SUCCESS RESPONSE
        return Response(
            {
                "message": (
                    f"{doctype.capitalize()} PDF "
                    "uploaded successfully."
                ),
                "doctype": doctype,
                "filename": safe_name,
                "chunks": chunk_count
            },
            status=200
        )
    # WINDOWS FILE LOCK
    except PermissionError:
        return Response(
            {
                "error": (
                    f"The existing {doctype} PDF is locked "
                    "by Windows, OneDrive, or another "
                    "application. Please close the PDF "
                    "and try again."
                )
            },
            status=409
        )
    # FILE NOT FOUND
    except FileNotFoundError as e:
        return Response(
            {
                "error": str(e)
            },
            status=400
        )
    # INVALID DOCUMENT
    except ValueError as e:
        return Response(
            {
                "error": str(e)
            },
            status=400
        )
    # OTHER ERROR
    except Exception as e:
        print(
            "Upload/indexing error:",
            str(e)
        )
        return Response(
            {
                "error": "Failed to upload and process PDF.",
                "details": str(e)
            },
            status=500
        )
    # CLEAN TEMP FILE
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except PermissionError:
                pass
# CHAT
@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def chat(request):
    user = request.user
    question = request.data.get("question")
    raw_doctype = request.data.get("doctype")
    # 1. QUESTION VALIDATION
    if question is None:
        return Response(
            {
                "error": "Question is required."
            },
            status=400
        )
    question = question.strip()
    if not question:
        return Response(
            {
                "error": "Question cannot be empty."
            },
            status=400
        )
    # 2. DOCUMENT TYPE VALIDATION & NORMALIZATION
    if not raw_doctype:
        return Response(
            {
                "error": "Document type is required."
            },
            status=400
        )
    normalized_type = str(raw_doctype).lower().strip()
    if normalized_type in ["general", "product", "product_knowledge"]:
        doctype = "general"
    elif normalized_type in ["technical", "technical_knowledge", "support"]:
        doctype = "technical"
    else:
        doctype = normalized_type
    if doctype not in [
        "general",
        "technical"
    ]:
        return Response(
            {
                "error": (
                    "Invalid document type. "
                    "Use 'general' or 'technical'."
                )
            },
            status=400
        )
    # 3. GET USER ROLE
    user_role = get_user_role(user)
    if user_role is None:
        return Response(
            {
                "error": "User profile is not configured."
            },
            status=403
        )
    # 4. PERMISSION CHECK
    # SUPERADMIN
    if user_role == "superadmin":
        allowed_doctypes = [
            "general",
            "technical"
        ]
    # SUPPORT ADMIN
    elif user_role == "support_admin":
        allowed_doctypes = [
            "technical"
        ]
    # PRODUCT ADMIN
    elif user_role == "product_admin":
        allowed_doctypes = [
            "general"
        ]
    # GENERAL USER
    elif user_role == "general_user":
        allowed_doctypes = [
            "general"
        ]
    else:
        return Response(
            {
                "error": "Invalid user role."
            },
            status=403
        )
    # CHECK KNOWLEDGE BASE PERMISSION
    if doctype not in allowed_doctypes:
        return Response(
            {
                "error": (
                    "You do not have permission "
                    "to access this knowledge base."
                )
            },
            status=403
        )
    # 5. ASK RAG
    try:
        result = ask_rag(
            question=question,
            doctype=doctype,
            user_role=user_role
        )
    except Exception as e:
        print(
            "Chat/RAG error:",
            str(e)
        )
        return Response(
            {
                "error": "Failed to get answer.",
                "details": str(e)
            },
            status=500
        )
    # 6. RETURN RESPONSE
    return Response(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        },
        status=200
    )
