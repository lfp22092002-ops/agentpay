# Publishing AgentPay SDK to PyPI

## Setup: Trusted Publishing (One-Time)

PyPI trusted publishing lets GitHub Actions publish without API tokens.

### 1. Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Register with `hello@leofundmybot.dev` (or G's email)
3. Enable 2FA (required)

### 2. Reserve the Package Name
1. Go to https://pypi.org/manage/account/publishing/
2. Add a "pending publisher":
   - **PyPI project name**: `agentpay`
   - **Owner**: `lfp22092002-ops`
   - **Repository**: `agentpay`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
3. Click "Add"

### 3. Create GitHub Environment
1. Go to https://github.com/lfp22092002-ops/agentpay/settings/environments
2. Create environment named `pypi`
3. No secrets needed — trusted publishing uses OIDC

### 4. Create a Release
```bash
gh release create v0.1.0 --title "v0.1.0 — First Release" --notes-file CHANGELOG.md
```

This triggers `.github/workflows/publish.yml` which:
- Builds the SDK from `sdk/`
- Publishes to PyPI via trusted publishing (OIDC, no token)
- Publishes TS SDK to npm (needs `NPM_TOKEN` secret)

### 5. Verify
```bash
pip install agentpay
python -c "from agentpay import AgentPayClient; print('✅')"
```

## npm Publishing (Optional)

1. Create npm account at https://www.npmjs.com/signup
2. Generate access token: `npm token create`
3. Add to GitHub: Settings → Secrets → `NPM_TOKEN`
4. Release triggers npm publish automatically

## Manual Publish (Fallback)

```bash
cd sdk
python -m build
twine upload dist/*  # needs ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD
```
