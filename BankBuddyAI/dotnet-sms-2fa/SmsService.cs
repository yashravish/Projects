using System;
using Twilio;
using Twilio.Rest.Api.V2010.Account;
using Twilio.Types;

namespace Sms2FA
{
    public class SmsService
    {
        // Replace these dummy credentials with your actual Twilio account SID and Auth Token.
        private const string accountSid = "YOUR_TWILIO_ACCOUNT_SID";
        private const string authToken = "YOUR_TWILIO_AUTH_TOKEN";

        public SmsService()
        {
            TwilioClient.Init(accountSid, authToken);
        }

        public bool Send2FASms(string toPhoneNumber, string message)
        {
            try
            {
                var messageOptions = new CreateMessageOptions(
                    new PhoneNumber(toPhoneNumber));
                messageOptions.From = new PhoneNumber("YOUR_TWILIO_PHONE_NUMBER");
                messageOptions.Body = message;

                var msg = MessageResource.Create(messageOptions);
                Console.WriteLine("SMS sent: " + msg.Sid);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("SMS sending failed: " + ex.Message);
                return false;
            }
        }
    }
}
