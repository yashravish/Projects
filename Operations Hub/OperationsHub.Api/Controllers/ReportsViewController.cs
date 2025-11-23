using Microsoft.AspNetCore.Mvc;

namespace OperationsHub.Api.Controllers
{
    [Route("Reports")]
    public class ReportsViewController : Controller
    {
        [Route("Sales")]
        public IActionResult Sales()
        {
            return View("~/Views/Reports/Sales.cshtml");
        }

        [Route("InventoryAging")]
        public IActionResult InventoryAging()
        {
            return View("~/Views/Reports/InventoryAging.cshtml");
        }

        [Route("FillRate")]
        public IActionResult FillRate()
        {
            return View("~/Views/Reports/FillRate.cshtml");
        }
    }
}

