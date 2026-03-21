@preconcurrency import AVFoundation
import Speech

/// Converts audio buffers from the microphone's native format to the format
/// required by SpeechAnalyzer (typically 1ch, 16 kHz, Int16) and feeds them
/// into the analyzer's input stream.
///
/// All work is serialized on a dedicated DispatchQueue so we never block the
/// real-time audio render thread and avoid GCD queue-assertion traps.
final class AudioBufferProcessor: @unchecked Sendable {
    private let converter: AVAudioConverter?
    private let targetFormat: AVAudioFormat?
    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private let queue = DispatchQueue(
        label: "com.livetranscribe.audio-convert",
        qos: .userInteractive
    )

    /// - Parameters:
    ///   - from: The microphone's native format (e.g. 1ch Float32 48 kHz).
    ///   - to:   The format SpeechAnalyzer wants (e.g. 1ch Int16 16 kHz).
    ///           Pass `nil` to skip conversion (formats already match).
    init(
        from sourceFormat: AVAudioFormat,
        to targetFormat: AVAudioFormat?,
        continuation: AsyncStream<AnalyzerInput>.Continuation
    ) {
        self.targetFormat = targetFormat
        self.continuation = continuation

        if let targetFormat {
            self.converter = AVAudioConverter(from: sourceFormat, to: targetFormat)
        } else {
            self.converter = nil
        }
    }

    private var bufferCount = 0

    func process(_ buffer: AVAudioPCMBuffer) {
        queue.async { [self] in
            bufferCount += 1
            if bufferCount <= 3 {
                print("[DEBUG] AudioBufferProcessor: processing buffer #\(bufferCount) frames=\(buffer.frameLength)")
            }

            guard let converter, let targetFormat else {
                // No conversion needed
                continuation.yield(AnalyzerInput(buffer: buffer))
                return
            }

            // Calculate output frame count based on sample-rate ratio
            let ratio = targetFormat.sampleRate / buffer.format.sampleRate
            let outputFrames = AVAudioFrameCount(Double(buffer.frameLength) * ratio)
            guard outputFrames > 0,
                  let output = AVAudioPCMBuffer(
                      pcmFormat: targetFormat,
                      frameCapacity: outputFrames
                  )
            else { return }

            var error: NSError?
            var consumed = false
            let status = converter.convert(to: output, error: &error) { _, outStatus in
                if consumed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                consumed = true
                outStatus.pointee = .haveData
                return buffer
            }

            if status == .haveData || status == .inputRanDry {
                if bufferCount <= 3 {
                    print("[DEBUG] AudioBufferProcessor: yielding converted buffer #\(bufferCount) frames=\(output.frameLength)")
                }
                continuation.yield(AnalyzerInput(buffer: output))
                if bufferCount <= 3 {
                    print("[DEBUG] AudioBufferProcessor: yield completed for #\(bufferCount)")
                }
            }
        }
    }

    func finish() {
        queue.async { [self] in
            continuation.finish()
        }
    }
}
