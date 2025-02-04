using System;

namespace Sms2FA
{
    class Program
    {
        static void Main(string[] args)
        {
            var smsService = new SmsService();
            // Replace with the target phone number and a sample message.
            bool success = smsService.Send2FASms("+15551234567", "Your BankBuddy 2FA code is 123456");
            Console.WriteLine("2FA SMS sent successfully: " + success);
        }
    }
}
