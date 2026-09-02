# 远程 MCP 接入与授权使用

## 用户操作

1. 在 MCP Registry 或服务商官方文档确认 transport 为 `streamable-http`，复制 **MCP URL**；不要填写仓库 URL、官网 URL 或 Registry 条目 ID。
2. 在应用配置的“添加工具 → 自定义 → MCP 服务”中填写连接名称和 MCP HTTPS 地址。
3. 根据服务认证方式选择：
   - **OAuth 登录型**：凭据环境变量留空，点击“测试连接并安装”，浏览器会打开服务商登录与授权页；成功后工具会自动同步。
   - **Bearer/API Key 型**：先在部署后端的环境中设置密钥，再填写该环境变量的名称，例如 `ACME_MCP_TOKEN`。页面不会保存或回显密钥。
   - **无认证型**：凭据环境变量留空，工具会直接发现并安装。
4. 在“已安装”展开连接，确认工具已经同步，再把需要的工具添加到当前智能体。
5. 发起任务。读操作可按服务声明直接执行；未知或写操作会暂停，展示服务名、工具名和参数摘要。批准后才会真实调用；拒绝后智能体会说明未完成的原因并收尾。

## Bird 示例

- 连接名称：`Bird`
- MCP HTTPS 地址：`https://mcp.bird.com`
- Bearer 凭据环境变量：留空

Bird 使用浏览器 OAuth，不应填写 `BIRD_API_KEY`。首次点击安装后完成 Bird 登录；建议先调用 `whoami` 等只读工具验证，再进行邮件发送。发送邮件等具有副作用的 MCP 工具仍会进入逐次审批。

## 部署要求

- 安装 `requirements.txt` 中的 `cryptography`；OAuth access token、refresh token 和 PKCE verifier 仅以加密形式保存在 `runs/mcp_oauth`（可用 `MCP_OAUTH_ROOT` 改路径）。
- 生产环境必须设置强随机 `MCP_SECRET_KEY`，并把 `PUBLIC_API_BASE_URL` 设为用户浏览器可访问的后端基址，例如 `https://agent.example.com`。OAuth 回调为 `/api/tool-connections/oauth/callback`。
- 当前自定义连接只支持 HTTPS Remote MCP。`npx`、`uvx`、Docker 和本地 stdio 应由单独的本地连接器运行，不能把命令粘进 HTTPS 地址栏。

## 故障定位

- **401 / 等待登录**：URL 可访问，但服务要求 OAuth；点击“登录授权”，不要改成随意的环境变量名。
- **环境变量未配置或凭据无效**：Bearer 型服务找不到对应后端变量或变量已失效；在服务端更新变量后同步。
- **授权后仍没有工具**：在“已安装”点击“授权完成后同步工具”；如仍失败，检查 OAuth 回调地址是否等于部署时公开的 API 地址。
- **工具被拒绝**：没有发生外部写入。重新发起并批准，或要求智能体使用不需要该权限的替代方案。
