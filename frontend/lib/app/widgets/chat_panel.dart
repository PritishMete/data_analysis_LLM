import 'package:flutter/material.dart';
import 'package:insightflow/app/insightflow_controller.dart';

class ChatPanel extends StatelessWidget {
  final List<ChatMessage> history;
  final bool busy;
  final ValueChanged<String> onSubmit;
  final VoidCallback onClear;

  const ChatPanel({
    super.key,
    required this.history,
    required this.busy,
    required this.onSubmit,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text('AI Chat', style: Theme.of(context).textTheme.titleLarge),
                ),
                TextButton(onPressed: onClear, child: const Text('Clear')),
              ],
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView.builder(
                itemCount: history.length,
                itemBuilder: (context, index) {
                  final message = history[index];
                  final isUser = message.role == ChatRole.user;
                  return Align(
                    alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.symmetric(vertical: 4),
                      padding: const EdgeInsets.all(10),
                      constraints: const BoxConstraints(maxWidth: 360),
                      decoration: BoxDecoration(
                        color: isUser ? Theme.of(context).colorScheme.primaryContainer : Theme.of(context).colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(message.text),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 8),
            _Composer(busy: busy, onSubmit: onSubmit),
          ],
        ),
      ),
    );
  }
}

class _Composer extends StatefulWidget {
  final bool busy;
  final ValueChanged<String> onSubmit;

  const _Composer({
    required this.busy,
    required this.onSubmit,
  });

  @override
  State<_Composer> createState() => _ComposerState();
}

class _ComposerState extends State<_Composer> {
  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _send() {
    final text = _controller.text.trim();
    if (text.isEmpty || widget.busy) {
      return;
    }
    widget.onSubmit(text);
    _controller.clear();
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _controller,
            minLines: 1,
            maxLines: 3,
            textInputAction: TextInputAction.send,
            onSubmitted: (_) => _send(),
            decoration: const InputDecoration(
              hintText: 'Ask InsightFlow to filter, normalize, pivot, or chart locally.',
              border: OutlineInputBorder(),
            ),
          ),
        ),
        const SizedBox(width: 8),
        FilledButton(
          onPressed: widget.busy ? null : _send,
          child: const Text('Send'),
        ),
      ],
    );
  }
}
