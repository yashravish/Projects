from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Escrow
from .escrow_logic import process_escrow

@api_view(['POST'])
def create_escrow(request):
    """
    Create a new escrow record and process it.
    Expects JSON data with 'escrow_id' and 'amount'.
    """
    data = request.data
    escrow = Escrow.objects.create(
        escrow_id=data.get('escrow_id'),
        amount=data.get('amount')
    )
    new_status = process_escrow(escrow)
    escrow.status = new_status
    escrow.save()
    return Response({
        'escrow_id': escrow.escrow_id,
        'status': escrow.status
    }, status=status.HTTP_201_CREATED)
