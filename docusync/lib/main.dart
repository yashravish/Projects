import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
// import 'package:http/http.dart' as http; // Uncomment and configure when integrating with the backend

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(CollaborationApp());
}

class CollaborationApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DocuSync',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: DocumentEditorPage(documentId: 'doc1'),
    );
  }
}

class DocumentEditorPage extends StatefulWidget {
  final String documentId;
  DocumentEditorPage({required this.documentId});

  @override
  _DocumentEditorPageState createState() => _DocumentEditorPageState();
}

class _DocumentEditorPageState extends State<DocumentEditorPage> {
  final DatabaseReference _docRef =
      FirebaseDatabase.instance.ref().child('documents');
  late TextEditingController _controller;
  String _documentContent = "";
  bool _isPreview = false;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController();

    // Listen for real-time updates from Firebase Realtime Database.
    _docRef.child(widget.documentId).onValue.listen((event) {
      final data = event.snapshot.value as String?;
      if (data != null && data != _documentContent) {
        setState(() {
          _documentContent = data;
          _controller.text = data;
        });
      }
    });
  }

  void _updateDocument(String text) {
    // Update the document in Firebase.
    // In a full OT implementation, you would send operations rather than replacing the whole text.
    _docRef.child(widget.documentId).set(text);
  }

  // Function to simulate committing the current version to version history via backend.
  Future<void> _commitVersion() async {
    // In production, you would send an HTTP POST request to your Google App Engine backend.
    // Example using the http package:
    //
    // final response = await http.post(
    //   Uri.parse('https://your-app-engine-url/commit'),
    //   headers: {'Content-Type': 'application/json'},
    //   body: jsonEncode({
    //     'documentId': widget.documentId,
    //     'content': _controller.text,
    //   }),
    // );
    //
    // if (response.statusCode == 200) {
    //   // Handle success
    // } else {
    //   // Handle error
    // }
    print("Committed version: ${_controller.text}");
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Version committed successfully!')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('DocuSync'),
        actions: [
          IconButton(
            icon: Icon(Icons.save),
            tooltip: 'Commit Version',
            onPressed: _commitVersion,
          ),
          IconButton(
            icon: Icon(_isPreview ? Icons.edit : Icons.remove_red_eye),
            tooltip: _isPreview ? 'Edit Mode' : 'Preview Mode',
            onPressed: () {
              setState(() {
                _isPreview = !_isPreview;
              });
            },
          ),
        ],
      ),
      body: _isPreview
          ? Markdown(
              data: _controller.text,
              padding: EdgeInsets.all(16.0),
            )
          : Padding(
              padding: EdgeInsets.all(16.0),
              child: TextField(
                controller: _controller,
                maxLines: null,
                decoration: InputDecoration(
                  hintText: 'Enter Markdown text...',
                  border: OutlineInputBorder(),
                ),
                onChanged: (text) {
                  _updateDocument(text);
                },
              ),
            ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
