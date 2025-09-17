#!/bin/bash
# Sync script to update GPT-files repository with latest trading system files

set -e

echo "🔄 Syncing trading system files to GPT-files repository..."

# Define source and destination
SOURCE_DIR="/home/karson"
DEST_DIR="/home/karson/gpt-files-repo"

# Core Python files to sync
CORE_FILES=(
    "run_adaptive_trading.py"
    "market_data_ibkr_adapter.py"
    "feature_engineering.py"
    "enhanced_trading_environment.py"
    "sac_trading_agent.py"
    "recurrent_ppo_agent.py"
    "algorithm_selector.py"
    "paper_trading_executor.py"
    "phase3_enhanced_system.py"
    "ensemble_rl_coordinator.py"
    "online_learning_system.py"
    "meta_learning_selector.py"
    "lightgbm_signal_validator.py"
    "jax_advanced_features.py"
    "jax_technical_indicators.py"
    "demo_complete_trading_intelligence.py"
    "verify_trading_intelligence.py"
    "test_production_readiness.py"
    "advanced_risk_management.py"
    "comprehensive_system_monitor.py"
    "market_data_config.py"
    "memory_management_system.py"
    "error_handling_system.py"
    "system_optimization_config.py"
    "audit_logging_utils.py"
    "neural_feature_extractor.py"
    "rl_trading_pipeline.py"
    "news_ingestion_marketaux.py"
    "news_ingestion_ibkr.py"
    "news_data_utils.py"
    "ibkr_paper_broker.py"
    "order_management_system.py"
    "CLAUDE.md"
    "requirements.txt"
)

# Sync core files
echo "📋 Syncing core Python files..."
for file in "${CORE_FILES[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        cp -u "$SOURCE_DIR/$file" "$DEST_DIR/$file" 2>/dev/null && echo "  ✓ Updated: $file" || echo "  - No changes: $file"
    else
        echo "  ⚠ Missing: $file"
    fi
done

# Sync directories
echo "📁 Syncing directories..."

# Orders directory (excluding backups and cache)
if [ -d "$SOURCE_DIR/orders" ]; then
    rsync -av --exclude="*.bak.*" --exclude="__pycache__" "$SOURCE_DIR/orders/" "$DEST_DIR/orders/"
    echo "  ✓ Synced: orders/"
fi

# Tools directory
if [ -d "$SOURCE_DIR/tools" ]; then
    rsync -av "$SOURCE_DIR/tools/" "$DEST_DIR/tools/"
    echo "  ✓ Synced: tools/"
fi

# Tests directory
if [ -d "$SOURCE_DIR/tests" ]; then
    rsync -av "$SOURCE_DIR/tests/" "$DEST_DIR/tests/"
    echo "  ✓ Synced: tests/"
fi

# Configs directory
if [ -d "$SOURCE_DIR/configs" ]; then
    rsync -av "$SOURCE_DIR/configs/" "$DEST_DIR/configs/"
    echo "  ✓ Synced: configs/"
fi

# Bin directory (important scripts)
if [ -f "$SOURCE_DIR/bin/restart_ibgw_and_rearm.sh" ]; then
    mkdir -p "$DEST_DIR/bin"
    cp -u "$SOURCE_DIR/bin/restart_ibgw_and_rearm.sh" "$DEST_DIR/bin/"
    echo "  ✓ Synced: bin/restart_ibgw_and_rearm.sh"
fi

# Check for changes
cd "$DEST_DIR"
if git diff --quiet && git diff --cached --quiet; then
    echo "✅ No changes detected - repository is up to date"
    exit 0
fi

# Show what changed
echo ""
echo "📝 Changes detected:"
git status --short

# Optional: Auto-commit and push (uncomment to enable)
# read -p "Do you want to commit and push these changes? (y/n) " -n 1 -r
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     git add -A
#     git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
#     git push origin main
#     echo "✅ Changes pushed to GitHub"
# else
#     echo "ℹ️  Changes staged but not committed. Run 'git commit' when ready."
# fi

echo ""
echo "🎯 Sync complete! To commit changes:"
echo "  cd $DEST_DIR"
echo "  git add -A"
echo "  git commit -m 'Sync updates from main system'"
echo "  git push origin main"