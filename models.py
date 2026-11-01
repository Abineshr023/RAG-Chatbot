# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    ROLE_CHOICES = (
        ('support', 'Support'),
        ('product', 'Product'),
        ('general', 'General'),
    )

    SUPPORT_LEVEL_CHOICES = (
    ("L1", "L1"),
    ("L2", "L2"),
    ("L3", "L3"),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    support_level = models.CharField(max_length=20, choices=SUPPORT_LEVEL_CHOICES,null=True, blank=True)
    def __str__(self):
        if self.support_level:
            return(
                f"{self.user.username} - "
                f"{self.role} - "
                f"{self.support_level}"
            )
        return f"{self.user.username} - {self.role}"
