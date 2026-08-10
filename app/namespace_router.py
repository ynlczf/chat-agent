def assistant_namespace(rt):
    if rt.server_info:
        return (rt.server_info.assistant_id,)
    return ("local-agent",)


def user_namespace(rt):
    if rt.server_info and rt.server_info.user:
        return (rt.server_info.user.identity,)
    user_id = getattr(rt.context, "user_id", "local-user")
    return (user_id,)


def org_namespace(rt):
    org_id = getattr(rt.context, "org_id", "default-org")
    return (org_id,)


def def_namespace(rt):
    return ("default",)
