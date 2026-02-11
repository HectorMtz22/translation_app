import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import '../services/deepl_service.dart';

enum AppState { idle, listening, translating }

class TranslationScreen extends StatefulWidget {
  const TranslationScreen({super.key});

  @override
  State<TranslationScreen> createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  final stt.SpeechToText _speech = stt.SpeechToText();
  final DeepLService _deepL = DeepLService();

  AppState _appState = AppState.idle;
  bool _speechAvailable = false;
  String _koreanText = '';
  String _spanishText = '';
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initSpeech();
  }

  Future<void> _initSpeech() async {
    _speechAvailable = await _speech.initialize(
      onError: (error) {
        setState(() {
          _appState = AppState.idle;
          if (error.errorMsg != 'error_speech_timeout') {
            _errorMessage = 'Speech error: ${error.errorMsg}';
          }
        });
      },
      onStatus: (status) {
        if (status == 'done' || status == 'notListening') {
          if (_appState == AppState.listening) {
            setState(() => _appState = AppState.idle);
          }
        }
      },
    );
    setState(() {});
  }

  void _toggleListening() {
    if (_appState == AppState.listening) {
      _stopListening();
    } else {
      _startListening();
    }
  }

  void _startListening() {
    if (!_speechAvailable) {
      setState(() => _errorMessage = 'Speech recognition not available');
      return;
    }

    setState(() {
      _appState = AppState.listening;
      _errorMessage = null;
      _koreanText = '';
      _spanishText = '';
    });

    _speech.listen(
      localeId: 'ko-KR',
      onResult: (result) {
        setState(() {
          _koreanText = result.recognizedWords;
        });

        if (result.finalResult && _koreanText.isNotEmpty) {
          _translateText(_koreanText);
        }
      },
      listenOptions: stt.SpeechListenOptions(
        partialResults: true,
        cancelOnError: true,
      ),
    );
  }

  void _stopListening() {
    _speech.stop();
    setState(() => _appState = AppState.idle);
  }

  Future<void> _translateText(String text) async {
    setState(() => _appState = AppState.translating);

    try {
      final translation = await _deepL.translate(text);
      setState(() {
        _spanishText = translation;
        _appState = AppState.idle;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Translation failed: $e';
        _appState = AppState.idle;
      });
    }
  }

  void _copyTranslation() {
    if (_spanishText.isNotEmpty) {
      Clipboard.setData(ClipboardData(text: _spanishText));
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Copied to clipboard'),
          duration: Duration(seconds: 1),
        ),
      );
    }
  }

  String get _statusText {
    switch (_appState) {
      case AppState.idle:
        return 'Tap the microphone to start';
      case AppState.listening:
        return 'Listening...';
      case AppState.translating:
        return 'Translating...';
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Korean to Spanish'),
        backgroundColor: theme.colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // Korean transcription area
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest.withAlpha(80),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: _appState == AppState.listening
                        ? theme.colorScheme.primary
                        : theme.colorScheme.outline.withAlpha(50),
                    width: _appState == AppState.listening ? 2 : 1,
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Korean',
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: theme.colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Expanded(
                      child: SingleChildScrollView(
                        child: Text(
                          _koreanText.isEmpty ? '...' : _koreanText,
                          style: theme.textTheme.titleLarge?.copyWith(
                            color: _koreanText.isEmpty
                                ? theme.colorScheme.onSurface.withAlpha(100)
                                : null,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Status indicator
            Text(
              _statusText,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurface.withAlpha(160),
              ),
            ),

            // Error message
            if (_errorMessage != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  _errorMessage!,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),

            const SizedBox(height: 16),

            // Microphone button
            GestureDetector(
              onTap: _appState == AppState.translating ? null : _toggleListening,
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _appState == AppState.listening
                      ? theme.colorScheme.error
                      : theme.colorScheme.primary,
                  boxShadow: [
                    BoxShadow(
                      color: (_appState == AppState.listening
                              ? theme.colorScheme.error
                              : theme.colorScheme.primary)
                          .withAlpha(80),
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Icon(
                  _appState == AppState.listening ? Icons.stop : Icons.mic,
                  color: Colors.white,
                  size: 36,
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Spanish translation area
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest.withAlpha(80),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: theme.colorScheme.outline.withAlpha(50),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Spanish',
                          style: theme.textTheme.labelLarge?.copyWith(
                            color: theme.colorScheme.primary,
                          ),
                        ),
                        if (_spanishText.isNotEmpty)
                          IconButton(
                            icon: const Icon(Icons.copy, size: 20),
                            onPressed: _copyTranslation,
                            tooltip: 'Copy translation',
                          ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Expanded(
                      child: SingleChildScrollView(
                        child: _appState == AppState.translating
                            ? const Center(
                                child: CircularProgressIndicator(),
                              )
                            : Text(
                                _spanishText.isEmpty ? '...' : _spanishText,
                                style: theme.textTheme.titleLarge?.copyWith(
                                  color: _spanishText.isEmpty
                                      ? theme.colorScheme.onSurface.withAlpha(100)
                                      : null,
                                ),
                              ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
