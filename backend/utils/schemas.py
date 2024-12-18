from marshmallow import Schema, fields, EXCLUDE
from marshmallow.validate import Range

### DATA Schema
class BucketSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    id = fields.Integer(required=True)
    # TODO: confirm and enable
    title = fields.String(required=True)
    # description = fields.String(required=True)
    priority = fields.Int(required=True)


###########################

class MessageSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    sender = fields.String(required=True)
    message = fields.String(required=True)
    intent = fields.String(required=False)

class ChatRequestSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    messages = fields.List(fields.Nested(MessageSchema), required=True)
    message_id = fields.Integer(required=True, validate=Range(min=-1))
    host_url = fields.String(required=True)
    prompt = fields.String(required=True)
    pinecone_index = fields.String(required=True)
    namespace = fields.String(required=True)
    closure_msg = fields.String(missing="Is there anything else I can assist you with?")
    conversation_status = fields.String()
    org_description = fields.String()
    # optional fields
    unsure_msg = fields.String(
        allow_none=True,
        missing="I'm sorry, I'm not sure how to respond to that. Can you please rephrase?",
    )
    sender_country = fields.String(
        allow_none=True,
    )
    sender_city = fields.String(
        allow_none=True,
    )
    filters = fields.Dict()
    buckets = fields.List(fields.Nested(BucketSchema))


class FetchVectorsSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    messages = fields.List(fields.Nested(MessageSchema), required=True)
    host_url = fields.String(required=True)
    pinecone_index = fields.String(required=True)
    message_id = fields.Integer(required=True, validate=Range(min=1))
    namespace = fields.String(required=True)
