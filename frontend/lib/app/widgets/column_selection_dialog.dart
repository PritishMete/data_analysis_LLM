import 'package:flutter/material.dart';

class ColumnChoice {
  final String columnId;
  final String label;

  const ColumnChoice({
    required this.columnId,
    required this.label,
  });
}

class ColumnSelectionDialog extends StatelessWidget {
  final String targetLabel;
  final List<ColumnChoice> choices;
  final ValueChanged<String> onSelected;
  final VoidCallback onCancel;

  const ColumnSelectionDialog({
    super.key,
    required this.targetLabel,
    required this.choices,
    required this.onSelected,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Select a column'),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420, maxHeight: 420),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('I could not map "$targetLabel" safely. Pick the correct column.'),
            const SizedBox(height: 12),
            Flexible(
              child: ListView.separated(
                shrinkWrap: true,
                itemCount: choices.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (context, index) {
                  final choice = choices[index];
                  return ListTile(
                    dense: true,
                    title: Text(choice.label),
                    subtitle: Text(choice.columnId),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => onSelected(choice.columnId),
                  );
                },
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(onPressed: onCancel, child: const Text('Cancel')),
      ],
    );
  }
}
