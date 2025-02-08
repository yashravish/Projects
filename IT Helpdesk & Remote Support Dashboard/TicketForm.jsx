// TicketForm.jsx
function TicketForm() {
    const [ticket, setTicket] = useState({
      title: '',
      description: '',
      priority: 'Medium'
    });
  
    const handleSubmit = async () => {
      await axios.post('/api/tickets', ticket);
      // Refresh ticket list
    };
  }