// ComplianceHeatmap.jsx
function ComplianceHeatmap() {
    const [data, setData] = useState([]);
  
    useEffect(() => {
      axios.get('/api/compliance/stats').then(res => {
        setData(res.data);
      });
    }, []);
  
    return (
      <HeatMap
        data={data}
        xField="department"
        yField="check_type"
        colorField="status"
      />
    );
  }