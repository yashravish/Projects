// api/Controllers/DocumentController.cs
using Microsoft.AspNetCore.Mvc;

namespace DocuVaultChainAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class DocumentController : ControllerBase
    {
        // GET api/document/{id}
        [HttpGet("{id}")]
        public IActionResult GetDocument(string id)
        {
            // In a real system, validate the user's role and retrieve the document securely.
            // For demonstration, return a dummy document.
            return Ok(new 
            { 
                DocumentId = id, 
                Content = "Securely retrieved document content.",
                Metadata = "Audit-ready metadata" 
            });
        }
    }
}
