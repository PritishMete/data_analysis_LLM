import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:insightflow/models/operation_plan.dart';

class PlanRequestPayload {
  final String requestId;
  final String requestText;
  final String locale;
  final String clientVersion;

  const PlanRequestPayload({
    required this.requestId,
    required this.requestText,
    required this.locale,
    required this.clientVersion
  });

  Map<String, dynamic> toJson() => {
    'request_id': requestId,
    'request_text': requestText,
    'locale': locale,
    'client_version': clientVersion
  };
}

class PlanResponseEnvelope {
  final bool ok;
  final String requestId;
  final InsightFlowPlan? plan;
  final String? errorCode;
  final String? errorMessage;
  final bool repaired;

  const PlanResponseEnvelope({
    required this.ok,
    required this.requestId,
    required this.plan,
    required this.errorCode,
    required this.errorMessage,
    required this.repaired
  });

  factory PlanResponseEnvelope.fromJson(Map<String, dynamic> json) {
    final ok = json['ok'] as bool;
    if (ok) {
      return PlanResponseEnvelope(
        ok: true,
        requestId: json['request_id'] as String,
        plan: InsightFlowPlan.fromJson(json['plan'] as Map<String, dynamic>),
        errorCode: null,
        errorMessage: null,
        repaired: json['repaired'] as bool? ?? false
      );
    }
    final error = json['error'] as Map<String, dynamic>;
    return PlanResponseEnvelope(
      ok: false,
      requestId: json['request_id'] as String,
      plan: null,
      errorCode: error['code'] as String,
      errorMessage: error['message'] as String,
      repaired: false
    );
  }
}

class PlanClient {
  final Uri endpoint;
  final http.Client _client;

  PlanClient({required this.endpoint, http.Client? client}) : _client = client ?? http.Client();

  Future<PlanResponseEnvelope> requestPlan(PlanRequestPayload payload) async {
    final response = await _client.post(
      endpoint,
      headers: {
        'content-type': 'application/json'
      },
      body: jsonEncode(payload.toJson())
    );
    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return PlanResponseEnvelope.fromJson(decoded);
  }
}
