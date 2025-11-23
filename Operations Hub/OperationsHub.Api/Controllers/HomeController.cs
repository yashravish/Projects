using Microsoft.AspNetCore.Mvc;

namespace OperationsHub.Api.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}

