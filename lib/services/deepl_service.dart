import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';

class DeepLService {
  static const _baseUrl = 'https://api-free.deepl.com/v2/translate';

  String get _apiKey => dotenv.env['DEEPL_API_KEY'] ?? '';

  Future<String> translate(String text) async {
    if (text.trim().isEmpty) return '';

    final response = await http.post(
      Uri.parse(_baseUrl),
      headers: {
        'Authorization': 'DeepL-Auth-Key $_apiKey',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'text': [text],
        'source_lang': 'KO',
        'target_lang': 'ES',
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['translations'][0]['text'] as String;
    } else {
      throw Exception('DeepL API error: ${response.statusCode} ${response.body}');
    }
  }
}
